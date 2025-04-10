# 说明

在使用 docker compose 的时候, 由于有内部网络, `.env` 文件里的 `REDIS_URL` 必须与 `docker-compose` 里的 services 的名字一致, `REDIS_PORT` 可以任意修改

```bash
# 在项目根目录下执行下面
# 单独打打镜像
docker build -f docker/app.dockerfile -t myenv_app .

docker compose -f docker/docker-compose.yml --env-file .env up   # .env 默认用 docker-compose.yml 同级目录下的, 但我们一般需要把它放在项目目录下
```

TODO: REDIS_URL 与 docker-compose.yml 一致稍显诡异, 能否优化