package api

import (
	"encoding/json"
	"net/http"
	"bufio"
	"os/exec"
)

type AnalizeType struct {
	Link 		string
	Task 		string
	Sensor 	string
}



func Calculate(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	var analize AnalizeType

	json.NewDecoder(request.Body).Decode(&analize)


	cmd := exec.Command("python3", "pythonfile.py", analize.Link, analize.Task, analize.Sensor)

  stdout, err := cmd.StdoutPipe()
  if err != nil {
    panic(err)
	}
	
  err = cmd.Start()
  if err != nil {
    panic(err)
	}
	
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
			json.NewEncoder(response).Encode(scanner.Text())	
	}

	cmd.Wait()
}