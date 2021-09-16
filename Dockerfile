FROM continuumio/miniconda3

MAINTAINER Shubhanshu Mishra <smishra@twitter.com>

ARG conda_env=image-crop-analysis

COPY . ./
WORKDIR .

RUN apt-get update --fix-missing && apt-get install -y zip vim gcc g++ make
RUN conda update conda --yes
RUN conda install pip --yes
RUN pip install pandas scikit-learn scikit-image statsmodels requests dash jupyterlab streamlit pillow six ninja ipykernel matplotlib
# CMD ["jupyter", "lab", "--port=8900", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
CMD streamlit run --server.port 8080 --server.enableCORS false notebooks/app.py
