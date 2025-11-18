# Base Windows image
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Install Python 3.10
ADD https://www.python.org/ftp/python/3.10.8/python-3.10.8-amd64.exe C:\\python-installer.exe
RUN C:\\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 && del C:\\python-installer.exe

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

