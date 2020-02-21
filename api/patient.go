package api

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	_ "github.com/lib/pq"

	"../contants"
)

const (
	host     = "46.101.116.184"
	port     = 5432
	user     = "nivolves"
	password = "14881488"
	dbname   = "hospital"
)

type Patient struct {
	PatientID   int
	DoctorID    int
	Age					int
	Height      float32
	Weight      float32
	FirstName   string
	LastName    string
	FathersName string
	Diagnosis   string
}

func AddPatient(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	var patient Patient
	var patientId int

	var _ = json.NewDecoder(request.Body).Decode(&patient)

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	sql := `
	INSERT INTO patients (doctorID, height, weight, firstName, lastName, fathersName, diagnosis, age)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	RETURNING patientID`

	err = db.QueryRow(sql, patient.DoctorID, patient.Height, patient.Weight, patient.FirstName, patient.LastName, patient.FathersName, patient.Diagnosis, patient.Age).Scan(&patientId)
	if err != nil {
		panic(err)
	}

	patient.PatientID = patientId

	json.NewEncoder(response).Encode(&patient)
}

func GetPatients(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	var patient Patient
	var patients []Patient

	rows, err := db.Query("SELECT * FROM patients")

	if err != nil {
		log.Fatal(err)
	}

	for rows.Next() {
		if err := rows.Scan(&patient.PatientID, &patient.DoctorID, &patient.FirstName, &patient.LastName, &patient.FathersName, &patient.Height, &patient.Weight, &patient.Diagnosis, &patient.Age); err != nil {
			log.Fatal(err)
		}
		patients = append(patients, patient)
	}
	if err := rows.Err(); err != nil {
		log.Fatal(err)
	}

	json.NewEncoder(response).Encode(patients)
}

func DeletePatient(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")

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
	DELETE FROM patients
	WHERE patientID = $1;`
	_, err = db.Exec(sqlStatement, id)
	if err != nil {
		panic(err)
	}
}

func UpdatePatient(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		contants.Host, contants.Port, contants.User, contants.Password, contants.Dbname)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	defer db.Close()

	var patient Patient

	var _ = json.NewDecoder(request.Body).Decode(&patient)

	params := mux.Vars(request)
	id := params["id"]

	sqlStatement := `
UPDATE patients
SET doctorID = $2, height = $3, weight = $4, firstName = $5, lastName = $6, fathersName = $7, diagnosis = $8, age = $9
WHERE patientID = $1;`
	_, err = db.Exec(sqlStatement, id, &patient.DoctorID, &patient.Height, &patient.Weight, &patient.FirstName, &patient.LastName, &patient.FathersName, &patient.Diagnosis, &patient.Age)
	if err != nil {
		panic(err)
	}
}
