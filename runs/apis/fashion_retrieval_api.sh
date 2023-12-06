#!/bin/zsh
uvicorn apis.app_polyvore:app --reload --host 127.0.0.1 --port 3000
