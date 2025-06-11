import express from 'express';
import bodyParser from 'body-parser';
import { initialize,processUserInput } from './app.js';
import dotenv from 'dotenv';

dotenv.config();

process.env.NODE_ENV = 'server';

const app = express();
const PORT = process.env.PORT;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.post('/q', async (req, res) => {
    const userInput = req.body.input;
    if(!userInput) {
        return res.status(400).json({ error: 'Input is required' });
    }

    try{
        await initialize();
        const response = await processUserInput(userInput);
        res.json(response);
    }catch(error){
        res.status(500).json({error : error.message});
        console.error('Error', error);
    }
});

app.listen(PORT,() => {
    console.log(`Server is running on port ${PORT}`);
});