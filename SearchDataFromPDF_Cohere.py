def  load_pdf(path):
        from PyPDF2 import PdfReader
        import tiktoken
        
        reader = PdfReader(path)
        parts = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            y = tm[5]
            if y > 50 and y < 720:
                parts.append(text)
        
        for page in reader.pages:
            # page = reader.pages[3]
            page.extract_text(visitor_text=visitor_body)
        
        text_body = "".join(parts)
        return text_body

if __name__=="__main__":
    text1=load_pdf('data/A.pdf')
    text2=load_pdf('data/B.pdf')
    text3=load_pdf('data/C.pdf')

    import cohere
    import numpy as np

    cohere_key = "your cohere key"   #Get your API key from www.cohere.com
    co = cohere.Client(cohere_key)

    docs = [text1,text2,text3]
    pdf_title=['A.pdf','B.pdf','C.pdf']

    #Encode your documents with input type 'search_document'
    doc_emb = co.embed(docs, input_type="search_document", model="embed-english-v3.0").embeddings
    doc_emb = np.asarray(doc_emb)


    #Encode your query with input type 'search_query'
    query = "What is embedding vector?"
    query_emb = co.embed([query], input_type="search_query", model="embed-english-v3.0").embeddings
    query_emb = np.asarray(query_emb)
    query_emb.shape

    #Compute the dot product between query embedding and document embedding
    scores = np.dot(query_emb, doc_emb.T)[0]

    #Find the highest scores
    max_idx = np.argsort(-scores)

    print(f"Query: {query}")
    for idx in max_idx:
        print(f"Score: {scores[idx]:.2f}")
        print(pdf_title[idx])
        print("--------")
