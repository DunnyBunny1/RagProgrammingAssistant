# use python as our base image
FROM python:3.12-slim

# set `/frontend_code` as our working directory, matching our volume mount point in `docker-compose.yml`
WORKDIR /frontend_code

# copy over our reqs file into img's working dir, and do not cache the requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# expose port 8501 (default streamlit port) from the container
EXPOSE 8501

# run the streamlit `app.py` (home page) file at runtime on port 8501, creating a streamlit HTTP host
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
