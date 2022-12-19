# Start from a cuda base image for enabling GPU execution
FROM pytorch/pytorch

# Copy sail-on code to the container
COPY . /sail-on

# Change working directory
WORKDIR /sail-on

RUN pip install -e .

ENTRYPOINT ["/bin/bash"]
