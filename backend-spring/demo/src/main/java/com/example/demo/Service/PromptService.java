package com.example.demo.Service;


import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class PromptService {
    

    public Optional<Object> sendPromptToLangChain(String prompt){
        RestTemplate template=new RestTemplate();
        String lanChainUrl="http://langchains:5000/q";

        HttpHeaders header=new HttpHeaders();
        header.setContentType(MediaType.APPLICATION_JSON);

        Map<String,String> body=new HashMap<>();

        body.put("input",prompt);

        HttpEntity<Map<String,String>>request=new HttpEntity<>(body,header);
        ResponseEntity<Object>reponse=template.postForEntity(lanChainUrl,request,Object.class);
        return Optional.of(reponse.getBody());
    }
}
