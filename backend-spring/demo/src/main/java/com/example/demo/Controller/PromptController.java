package com.example.demo.Controller;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.Service.PromptService;

@RestController
@RequestMapping("/api")
public class PromptController {
    
    private PromptService service;
    @Autowired
    public PromptController(PromptService service){
        this.service=service;
    }

    @PostMapping("/prompts")
    public ResponseEntity<Object> getResponse(@RequestBody Map<String,String> input){
        String prompt=input.get("input");
Optional<Object> result=service.sendPromptToLangChain(prompt);
if(result.isPresent()){
    return ResponseEntity.status(HttpStatus.OK).body(result);
}else{
return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
}
    }

}
