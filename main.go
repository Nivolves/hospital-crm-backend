package main

import (
	"net/http"
	"os"

	"github.com/gorilla/handlers"

	"./api"

	_ "image/jpeg"

	"github.com/gorilla/mux"
)

const (
	STATIC_DIR = "/assets/"
)

func StartServer() {

	router := mux.NewRouter()
	changeHeaderThenServe := func(h http.Handler) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			h.ServeHTTP(w, r)
		}
	}
	router.PathPrefix(STATIC_DIR).Handler(http.StripPrefix(STATIC_DIR, changeHeaderThenServe(http.FileServer(http.Dir("."+STATIC_DIR)))))

	router.HandleFunc("/image", api.AddImage).Methods("POST")
	router.HandleFunc("/patient", api.AddPatient).Methods("POST")
	router.HandleFunc("/analize", api.Analize).Methods("POST")
	router.HandleFunc("/patients", api.GetPatients).Methods("GET")
	router.HandleFunc("/images", api.GetImages).Methods("GET")
	router.HandleFunc("/patient/{id}", api.DeletePatient).Methods("DELETE")
	router.HandleFunc("/patient/{id}", api.UpdatePatient).Methods("PUT")
	http.ListenAndServe(":"+os.Getenv("PORT"), handlers.CORS()(router))
}


func main() {
	StartServer()
}
