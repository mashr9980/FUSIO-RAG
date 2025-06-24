import uvicorn
from main import RAGApplication  
from config import config

if __name__ == "__main__":
    # Initialize application
    rag_app = RAGApplication()
    
    # Configure and run server
    uvicorn_config = uvicorn.Config(
        app=rag_app.app,
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        ws_max_size=16777216,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        loop="auto",
        ws="websockets",
        log_level="info",
        ssl_certfile='certificate.crt',
        ssl_keyfile='private.key'
    )
    
    # Start server
    server = uvicorn.Server(uvicorn_config)
    server.run()