from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Je t'aime mon Alix, je n'aime pas quand t'es malade!! <3"}


@app.get("/new/{item_id}")
async def root_new(item_id: int):
    return {
        "message": f"Je t'aime mon Alix {item_id} fois plus, je n'aime pas quand t'es malade!! <3"
    }
