package api

import (
	"fmt"
	"encoding/json"
	"net/http"
	"bufio"
	"os/exec"
)

type AnalizeType struct {
	Link 		string
	Task 		string
	Sensor 	string
	SaveTransform string
	SaveBinarization string
}



func Calculate(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Access-Control-Allow-Origin", "*")
	response.Header().Set("content-type", "application/json")
	response.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	var analize AnalizeType

	json.NewDecoder(request.Body).Decode(&analize)


	cmd := exec.Command("python3", "SystemBack/pythonfile.py", analize.Link, analize.Task, analize.Sensor, analize.SaveTransform, analize.SaveBinarization)

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
			fmt.Print(scanner.Text())
			json.NewEncoder(response).Encode(scanner.Text())	
	}

	cmd.Wait()
}