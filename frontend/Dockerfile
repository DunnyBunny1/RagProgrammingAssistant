# use python as our base image
FROM python:3.12-slim

# set `/app` as our working directory, matching our volume mount point in `docker-compose.yml`
WORKDIR /app

# copy our entire project into the build
COPY . /app
# install the backend's reqiurements file
RUN pip install --no-cache-dir -r frontend/requirements.txt

# expose port 8501 (default streamlit port) from the container
EXPOSE 8501

# set the working directory to the frontend folder
WORKDIR /app/frontend/src

# run the streamlit `app.py` (home page) file at runtime on port 8501, creating a streamlit HTTP host
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
