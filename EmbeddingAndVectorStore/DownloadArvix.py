import arxiv
from tqdm import tqdm
import time
from urllib.error import HTTPError

class Arvix_download:

    def __init__(self,query):
       self.query=query

    def arxivQuery(self):
        # Construct the default API client.
        client = arxiv.Client()

        # Search for the 10 most recent articles matching the keyword "quantum."
        search = arxiv.Search(
        query = "large language model",
        max_results = 100,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
        results=client.results(search)

        all_results = list(results)
        return all_results

    def download_file(self,all_results):
        for idx,x in enumerate(all_results):
            x=str(x)
            id=x.split('/')[-1]
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[id])))
            f_name=f"{idx+1}.pdf"
            paper.download_pdf(dirpath="./arvix_pdfs", filename=f_name)
            print(f"downloaded {id}")

if __name__=="__main__":
  query='large language model'
  obj=Arvix_download(query)
  results=obj.arxivQuery()
  obj.download_file(results)
