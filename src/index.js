import fetch from 'node-fetch';
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
import { createClient } from '@supabase/supabase-js'
import {SupabaseVectorStore} from "langchain/vectorstores/supabase"
import {OpenAIEmbeddings} from "langchain/embeddings/openai"
require("dotenv").config()



async function splitText(url){

    const response = await fetch(url);
    const text = await response.text();
    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 500,
        chunkOverlap: 50,
        lengthFunction: (text) =>
        text.length,
        separators: ["\n\n", "\n", " ", ""],
    });
    const docs = await splitter.createDocuments([text]);
    // console.log(docs);
    const sbPUrl = process.env.SUPABASE_PROJECT_URL
    const sbApiKey = process.env.SUPABASE_API_KEY
    const openaiApikey = process.env.OPEN_AI_API_KEY
    const client = createClient(sbPUrl, sbApiKey)
    await SupabaseVectorStore.fromDocuments(
        docs,
        new OpenAIEmbeddings({openAIApiKey: openaiApikey}),
        {client: client,
        tableName: 'documents'
        }
    )

}

// splitText('https://www.gutenberg.org/files/1342/1342-0.txt')
splitText('https://github.com/abdoutony/langchain_course/blob/master/data/dataset.txt')
