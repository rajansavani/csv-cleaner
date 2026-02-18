import http from "k6/http";
import { check, sleep } from "k6";
import { open } from "k6/fs";

export const options = {
  vus: 3,
  duration: "30s",
};

export default function () {
  const url = "https://csvcleaner.onrender.com/clean/basic";
  const csv = open("./data/raw/messy_IMDB_dataset.csv", "b");

  const payload = {
    file: http.file(csv, "messy_IMDB_dataset.csv", "text/csv"),
  };

  const res = http.post(url, payload);
  check(res, { "status is 200": (r) => r.status === 200 });
  sleep(0.2);
}
