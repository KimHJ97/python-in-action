# FastAPI Cheat Sheet

## Poetry 가상 환경 사용 방법

```bash
# poetry 설치
pip install poetry

# poetry 초기화
poetry init

# poetry 가상 환경 접속
poetry shell

# 라이브러리 설치
poetry add {library name}
```

## FastAPI 프로젝트 만들기

```bash
poetry init
poetry shell
poetry add fastapi
poetry add "uvicorn[standard]"

# 프로젝트 실행
uvicorn main:app --reload --port 8080
```

## Docker 환경에 MySQL 설치

```bash
# MySQL 컨테이너 실행
docker run --name mysql-local -p 3306:3306/tcp -e MYSQL_ROOT_PASSWORD=test -d mysql:8.0

# MySQL 접속
docker exec -it mysql-local bash
mysql -u root -h 127.0.0.1 -p
Enter password: test

# 스키마 생성
create schema `fastapi-ca`;
show databases;

# 컨테이너 중지/시작/재시작
docker stop {Container-ID}
docker start {Container-ID}
docker restart {Container-ID}
```