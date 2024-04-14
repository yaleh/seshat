# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

RUN pip install --no-cache-dir poetry

# Copy all files from the current directory to the working directory in the 
# container
COPY . /app/

# Install the required dependencies from requirements.txt
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Expose the port (if necessary)
EXPOSE 7860

# Run the script when the container launches
CMD /usr/local/bin/python3 seshat.py