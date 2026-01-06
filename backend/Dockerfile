# use python as our base image
FROM python:3.12-slim

# set `/app` as our working directory, matching our volume mount point in `docker-compose.yml`
WORKDIR /app

# install dependencies: curl is needed for the health check for the API server
RUN apt-get update && apt-get install -y curl && apt-get clean

# copy our entire project into the build
COPY . /app

# install the backend's reqiurements file
RUN pip install --no-cache-dir -r backend/requirements.txt

# expose port 8000 (default fastAPI/Uvicorn port) from the container
EXPOSE 8000

# set the working directory to the backend folder
WORKDIR /app/backend/src

# run our FastAPI "app" (inside the "main" python module), creating an HTTP server with hot reloads
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]