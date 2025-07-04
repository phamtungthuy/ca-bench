/usr/sbin/nginx

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
export APP_HOST=${APP_HOST:-0.0.0.0}
export APP_PORT=${APP_PORT:-8000}


echo "Starting Nginx..."
exec uvicorn server.models_server:app \
  --host "$APP_HOST" \
  --port "$APP_PORT" 