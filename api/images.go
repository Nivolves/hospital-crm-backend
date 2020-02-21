package api

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"image/jpeg"
	_ "image/png"
	_ "golang.org/x/image/bmp"

	"../contants"
	"github.com/gorilla/mux"
)

type Image struct {
	ImageID   int
	PatientID int
	Name      string
	Type      string
	Date      time.Time
	Link      string
}

func (img *Image) createImage() string {

	reader := base64.NewDecoder(base64.StdEncoding, strings.NewReader(img.Link))
	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	filename, err := filepath.Abs("./assets/" + img.Name)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0777)
	if err != nil {
		log.Fatal(err)
	}

	err = jpeg.Encode(f, m, &jpeg.Options{Quality: 75})
	if err != nil {
		log.Fatal(err)
	}
	return filename
}

func AddImage(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/x-www-form-urlencoded")


	var image Image
	var ImageID int

	var _ = json.NewDecoder(request.Body).Decode(&image)

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	image.Date = time.Now()
	image.Link = image.createImage()

	sql := `
	INSERT INTO images (patientid, name, type, date)
	VALUES ($1, $2, $3, $4)
	RETURNING imageID`

	err = db.QueryRow(sql, image.PatientID, image.Name, image.Type, image.Date).Scan(&ImageID)
	if err != nil {
		panic(err)
	}

	image.ImageID = ImageID

	json.NewEncoder(response).Encode(&image)
}

func GetImages(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	var image Image
	var images []Image

	rows, err := db.Query("SELECT * FROM images")

	if err != nil {
		log.Fatal(err)
	}

	for rows.Next() {
		if err := rows.Scan(&image.ImageID, &image.PatientID, &image.Name, &image.Type, &image.Date); err != nil {
			log.Fatal(err)
		}
		images = append(images, image)
		image.Link, err = filepath.Abs("./assets/" + image.Name)
		if err != nil {
			log.Fatal(err)
		}
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}

	json.NewEncoder(response).Encode(images)
}

func DeleteImage(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	params := mux.Vars(request)
	id := params["id"]

	sqlStatement := `
	DELETE FROM images
	WHERE imageID = $1;`
	_, err = db.Exec(sqlStatement, id)
	if err != nil {
		panic(err)
	}
}
