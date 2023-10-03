#!/bin/sh
uvicorn apis.app_v2:app --reload --host 127.0.0.1 --port 3000
