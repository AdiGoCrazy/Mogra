"""ASGI Server Launcher for Mogra Movie Recommender Agent REST API."""

import argparse
import uvicorn

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mogra Movie Recommender Agent REST API Server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind server (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes.")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      🚀 MOGRA MOVIE RECOMMENDER AGENT — REST API SERVER LAUNCHED")
    print("=" * 80)
    print(f"  API Base URL            : http://{args.host}:{args.port}")
    print(f"  Interactive Swagger UI  : http://localhost:{args.port}/docs")
    print(f"  ReDoc Documentation     : http://localhost:{args.port}/redoc")
    print(f"  OpenAPI JSON Spec       : http://localhost:{args.port}/openapi.json")
    print("=" * 80 + "\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
