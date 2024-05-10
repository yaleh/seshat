# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

RUN pip install --no-cache-dir poetry

# Install the required dependencies from requirements.txt
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Copy files and folders from the current directory 
# to the working directory `/app` in the container:
COPY db/ /app/db/
COPY components/ /app/components/
COPY tools/ /app/tools/
COPY ui/ /app/ui/
COPY seshat.py config.yaml /app/

# Expose the port (if necessary)
EXPOSE 7860

# Run the script when the container launches
CMD /usr/local/bin/python3 seshat.py