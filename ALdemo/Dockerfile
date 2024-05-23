# pull latest julia image
FROM  julia:1.8.2

# create dedicated user
RUN useradd --create-home --shell /bin/bash genie

# set up the app
RUN mkdir /home/genie/app
COPY . /home/genie/app
WORKDIR /home/genie/app

# configure permissions
RUN chown -R genie:genie /home/

# switch user
USER genie

# instantiate Julia packages
RUN julia -e "using Pkg; Pkg.activate(\".\"); Pkg.instantiate(); Pkg.precompile(); "

# ports
EXPOSE 8000
EXPOSE 80

# set up app environment
ENV JULIA_DEPOT_PATH "/home/genie/.julia"
ENV GENIE_ENV "dev"
ENV GENIE_HOST "0.0.0.0"
ENV PORT "8000"
ENV WSPORT "8000"
ENV EARLYBIND "true"

# run app
#CMD ["bin/server"]

# or maybe include a Julia file
CMD julia -e 'using Pkg; Pkg.activate("."); using GenieFramework; Genie.loadapp();up(async=false);'
