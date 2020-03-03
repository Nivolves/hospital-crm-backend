package api

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"log"
	"time"

	_ "github.com/lib/pq"

	"../contants"
)


type Analize struct {
	AnalizeID   int
	PatientID   int
	Name   		  string
	Value 			string
	Date      time.Time
}

func AddAnalize(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	var analize Analize
	var analizeID int

	var _ = json.NewDecoder(request.Body).Decode(&analize)

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	analize.Date = time.Now()

	sql := `
	INSERT INTO analizes (patientID, name, value, date)
	VALUES ($1, $2, $3, $4)
	RETURNING analizeID`

	err = db.QueryRow(sql, analize.PatientID, analize.Name, analize.Value, analize.Date).Scan(&analizeID)
	if err != nil {
		panic(err)
	}

	analize.AnalizeID = analizeID

	json.NewEncoder(response).Encode(&analize)
}

func GetAnalizes(response http.ResponseWriter, request *http.Request) {
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

	var analize Analize
	var analizes []Analize

	rows, err := db.Query("SELECT * FROM analizes")

	if err != nil {
		log.Fatal(err)
	}

	for rows.Next() {
		if err := rows.Scan(&analize.AnalizeID, &analize.PatientID, &analize.Name, &analize.Value, &analize.Date); err != nil {
			log.Fatal(err)
		}
		analizes = append(analizes, analize)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}

	json.NewEncoder(response).Encode(analizes)
}