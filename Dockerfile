FROM xxx.cargo.io/release/solution-billtemplate:v0.5

RUN apt update && apt -y install git vim && \
    git clone -b clv_2.0 https://github.com/kuikuikuizzZ/OCR_solution.git /OCR_solution
WORKDIR /OCR_solution
RUN python setup.py  install build_ext --inplace
ENTRYPOINT python /OCR_solution/billTemplate/app.py
