FROM practice:latest

# Add code
ADD . /src

# Expose
EXPOSE 8002

#set the working directorly
WORKDIR /src

# RUN python3 manage.py runserver 0.0.0.0:8002