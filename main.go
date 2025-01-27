package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/qdrant/go-client/qdrant"
	"golang.org/x/sync/errgroup"
)

type Metadata struct {
	Data map[string]interface{} `json:"data"`
}

type QueryResult struct {
	Filename string                 `json:"filename"`
	Metadata map[string]interface{} `json:"metadata"`
	Score    float32                `json:"score"`
}

type Application struct {
	qdrantClient qdrant.Client
	httpClient   *http.Client
	jobQueue     chan func() error
}

func NewApplication() (*Application, error) {
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{
		Host:                   os.Getenv("QdrantHost"),
		APIKey:                 os.Getenv("QdrantClusterKey"),
		SkipCompatibilityCheck: true,
		UseTLS:                 true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Qdrant client: %w", err)
	}

	httpClient := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
			DisableKeepAlives:   false,
			MaxConnsPerHost:     20,
			MaxIdleConnsPerHost: 20,
		},
	}
	MaxConcurrentJobs, err := strconv.Atoi(os.Getenv("MaxConcurrentJobs"))

	if err != nil {
		return nil, fmt.Errorf("failed to parse MaxConcurrentJobs: %w", err)
	}

	jobQueue := make(chan func() error, MaxConcurrentJobs*2)
	g, _ := errgroup.WithContext(context.Background())
	for i := 0; i < MaxConcurrentJobs; i++ {
		g.Go(func() error {
			for job := range jobQueue {
				if err := job(); err != nil {
					fmt.Printf("Job failed: %v\n", err)
				}
			}
			return nil
		})
	}

	return &Application{
		qdrantClient: *qdrantClient,
		httpClient:   httpClient,
		jobQueue:     jobQueue,
	}, nil
}

func (app *Application) SetupCollections() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	exists, err := app.qdrantClient.CollectionExists(ctx, "images")
	if err != nil {
		return err
	}
	if exists {
		return nil
	}
	err = app.qdrantClient.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: "images",
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     768,
			Distance: qdrant.Distance_Cosine,
		}),
	})

	if err != nil {
		return err
	}
	return nil
}

func (app *Application) index(c echo.Context) error {
	// Process form data
	file, err := c.FormFile("image")
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "Failed to retrieve image")
	}

	metadataRaw := c.FormValue("metadata")
	if metadataRaw == "" {
		return echo.NewHTTPError(http.StatusBadRequest, "Metadata is required")
	}

	var metadata Metadata
	if err := json.Unmarshal([]byte(metadataRaw), &metadata.Data); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "Invalid metadata format")
	}

	data, err := readUploadedFile(file)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "Failed to read image")
	}

	app.jobQueue <- func() error {
		return app.processImage(data, file.Filename, metadata, c.Logger())
	}

	return c.JSON(http.StatusOK, map[string]string{
		"message": "Image processing started",
	})
}

func (app *Application) query(c echo.Context) error {

	file, err := c.FormFile("image")
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "Failed to retrieve image")
	}

	data, err := readUploadedFile(file)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "Failed to read image")
	}

	embedding, err := app.getEmbedding(data, file.Filename)
	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "Failed to get embedding: "+err.Error())
	}

	score_threshold := float32(0.5)

	searchResponse, err := app.qdrantClient.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "images",
		Query:          qdrant.NewQuery(embedding...),
		WithPayload:    qdrant.NewWithPayload(true),
		ScoreThreshold: &score_threshold,
	})

	if err != nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "Search failed: "+err.Error())
	}

	var results []QueryResult

	for _, result := range searchResponse {
		payload := result.Payload

		filenameVal, ok := payload["filename"]
		if !ok {
			continue
		}
		filename := filenameVal.GetStringValue()

		metadataVal, ok := payload["metadata"]
		if !ok {
			continue
		}

		metadataStruct := metadataVal.GetStructValue()
		metadata := convertQdrantPayload(metadataStruct.Fields)

		results = append(results, QueryResult{
			Filename: filename,
			Metadata: metadata,
			Score:    result.Score,
		})

	}

	return c.JSON(http.StatusOK, results)
}

func (app *Application) warmup(c echo.Context) error {
	app.jobQueue <- func() error {
		req, err := http.NewRequest("POST", os.Getenv("EmbeddingServiceURL"), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+os.Getenv("EmbeddingServiceToken"))
		_, err = app.httpClient.Do(req)

		if err != nil {
			return err
		}
		return nil
	}
	return c.String(http.StatusOK, "Warmup done")
}

func (app *Application) processImage(imgData []byte, filename string, metadata Metadata, log echo.Logger) error {
	embedding, err := app.getEmbedding(imgData, filename)
	if err != nil {
		return fmt.Errorf("embedding failed: %w", err)
	}

	point, err := createQdrantPoint(metadata, filename, embedding)
	if err != nil {
		return fmt.Errorf("failed to create point: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err = app.qdrantClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: "images",
		Points:         []*qdrant.PointStruct{point},
	})
	if err != nil {
		return fmt.Errorf("upsert failed: %w", err)
	}

	log.Info("Image added successfully")
	return nil
}

func (app *Application) getEmbedding(imgData []byte, filename string) ([]float32, error) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("image", filename)
	if err != nil {
		return nil, err
	}

	if _, err = part.Write(imgData); err != nil {
		return nil, err
	}
	writer.Close()

	req, err := http.NewRequest("POST", os.Getenv("EmbeddingServiceURL")+"/embed", body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+os.Getenv("EmbeddingServiceToken"))

	resp, err := app.httpClient.Do(req)

	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned %s", resp.Status)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if len(bodyBytes)%4 != 0 {
		return nil, fmt.Errorf("invalid byte length %d (not divisible by 4)", len(bodyBytes))
	}

	floatCount := len(bodyBytes) / 4
	embeddings := make([]float32, floatCount)

	err = binary.Read(bytes.NewReader(bodyBytes), binary.LittleEndian, &embeddings)
	if err != nil {
		return nil, err
	}

	return embeddings, nil
}

func convertQdrantPayload(qdrantPayload map[string]*qdrant.Value) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range qdrantPayload {
		switch val := v.Kind.(type) {
		case *qdrant.Value_StringValue:
			result[k] = val.StringValue
		case *qdrant.Value_DoubleValue:
			result[k] = val.DoubleValue
		case *qdrant.Value_BoolValue:
			result[k] = val.BoolValue
		case *qdrant.Value_IntegerValue:
			result[k] = val.IntegerValue
		default:
		}
	}
	return result
}

func readUploadedFile(file *multipart.FileHeader) ([]byte, error) {
	src, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer src.Close()

	return io.ReadAll(src)
}

func convertMetadata(data map[string]interface{}) map[string]*qdrant.Value {
	result := make(map[string]*qdrant.Value)
	for k, v := range data {
		switch val := v.(type) {
		case string:
			result[k] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: val}}
		case float64:
			result[k] = &qdrant.Value{Kind: &qdrant.Value_DoubleValue{DoubleValue: val}}
		case bool:
			result[k] = &qdrant.Value{Kind: &qdrant.Value_BoolValue{BoolValue: val}}
		default:
		}
	}
	return result
}

func createQdrantPoint(metadata Metadata, filename string, embedding []float32) (*qdrant.PointStruct, error) {
	metadataQD := convertMetadata(metadata.Data)

	return &qdrant.PointStruct{
		Id: &qdrant.PointId{
			PointIdOptions: &qdrant.PointId_Uuid{Uuid: uuid.New().String()},
		},
		Vectors: &qdrant.Vectors{
			VectorsOptions: &qdrant.Vectors_Vector{
				Vector: &qdrant.Vector{Data: embedding},
			},
		},
		Payload: map[string]*qdrant.Value{
			"metadata": {
				Kind: &qdrant.Value_StructValue{
					StructValue: &qdrant.Struct{Fields: metadataQD},
				},
			},
			"filename": {
				Kind: &qdrant.Value_StringValue{StringValue: filename},
			},
		},
	}, nil
}

func main() {
	app, err := NewApplication()
	if err != nil {
		fmt.Printf("Failed to initialize application: %v\n", err)
		os.Exit(1)
	}

	if err := app.SetupCollections(); err != nil {
		fmt.Printf("Failed to setup collections: %v\n", err)
		os.Exit(1)
	}

	e := echo.New()
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.POST("/index", app.index)
	e.POST("/query", app.query)
	e.GET("/warmup", app.warmup)
	port := ":" + os.Getenv("PORT")
	if err := e.Start(port); err != nil {
		fmt.Printf("Server error: %v\n", err)
		os.Exit(1)
	}
}
