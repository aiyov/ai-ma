from app.services.store import ProductStore

store = ProductStore()


def get_store() -> ProductStore:
    return store
