from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import io

from cloud_model import PXRFModel
from ebk_predict import dataframe_to_grid

app = FastAPI()

model = PXRFModel("pxrf_full_model.pkl")

@app.get("/")
def root():
    return {"status": "The ArchSight is now running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    result_df = model.predict(df)

    ebk_grid_df, original_points_df = (
    run_ebk_from_dataframe(result_df))

    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=ArchaeoSight_Results.csv"
        })
