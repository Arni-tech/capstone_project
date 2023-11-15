from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from function import ToolFinder  # Assuming ToolFinder is in a module named 'function'
import webbrowser

# Initialize FastAPI
app = FastAPI()

# Add CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can change this to a specific list of allowed origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the tool finder
tool_finder = ToolFinder('/home/arnav/capstone_project/imp_proj/data/vectorized_data.pkl',
                         '/home/arnav/capstone_project/imp_proj/data/usable_dataset1.csv')


# Define request/response models
class QueryRequest(BaseModel):
    query: str


class ToolInfoResponse(BaseModel):
    tool_names: List[str]
    tool_info: List[dict]


# Define API routes
@app.post("/find_tool/", response_model=ToolInfoResponse)
async def find_tool(query_request: QueryRequest):
    top_similar_tools = tool_finder.find_tool(query_request.query)
    tool_info = tool_finder.find_info(top_similar_tools)
    return {"tool_names": top_similar_tools, "tool_info": tool_info}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    url = "/home/arnav/capstone_project/imp_proj/src/index.html"
    webbrowser.get('google-chrome').open(url)    
    uvicorn.run(app, host="0.0.0.0", port=8000)
