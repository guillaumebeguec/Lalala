import { liveApi } from "./client";
import { mockApi } from "./mock";
import type { SearchApi } from "./types";

export const api: SearchApi = import.meta.env.VITE_API_MODE === "live" ? liveApi : mockApi;
export const API_MODE: "live" | "mock" = import.meta.env.VITE_API_MODE === "live" ? "live" : "mock";
