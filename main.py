from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import io

from gbdt_model_renewed import run_model

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    df = pd.read_csv(io.BytesIO(contents))

    result_df = run_model(df)

    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=archaeosight_results.csv"
        })