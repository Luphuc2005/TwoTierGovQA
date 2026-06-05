/**
 * Axios API client — handles base URL, auth headers, and token refresh.
 */
import axios from "axios";

const API_BASE = "/api";

const client = axios.create({
    baseURL: API_BASE,
    headers: { "Content-Type": "application/json" },
});

// ── Request interceptor: attach JWT token ────────────────────
client.interceptors.request.use((config) => {
    const token = localStorage.getItem("token");
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// ── Response interceptor: handle 401 ─────────────────────────
client.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem("token");
            // Only redirect if not already on auth pages
            if (
                !window.location.pathname.includes("/login") &&
                !window.location.pathname.includes("/register")
            ) {
                window.location.href = "/login";
            }
        }
        return Promise.reject(error);
    }
);

// ── Auth API ─────────────────────────────────────────────────

export const authAPI = {
    register: (email, password, fullName) =>
        client.post("/auth/register", { email, password, full_name: fullName }),

    login: (email, password) =>
        client.post("/auth/login", { email, password }),

    getMe: () => client.get("/auth/me"),
};

// ── Conversations API ────────────────────────────────────────

export const conversationAPI = {
    list: () => client.get("/conversations"),

    create: (title = "New Conversation") =>
        client.post("/conversations", { title }),

    get: (id) => client.get(`/conversations/${id}`),

    delete: (id) => client.delete(`/conversations/${id}`),
};

// ── Chat API ─────────────────────────────────────────────────

export const chatAPI = {
    getMessages: (conversationId) =>
        client.get(`/messages/${conversationId}`),

    sendMessage: (conversationId, message) =>
        client.post("/chat", { conversation_id: conversationId, message }),
};

// ── Upload API ───────────────────────────────────────────────

export const uploadAPI = {
    /** Upload a PDF file. Returns { doc_id, file_name, file_size, ... } */
    upload: (file) => {
        const formData = new FormData();
        formData.append("file", file);
        return client.post("/upload-document", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        });
    },

    /** Get the SSE stream URL for a document ingestion (includes auth token). */
    getStreamUrl: (docId) => {
        const token = localStorage.getItem("token");
        return `/api/upload-document/${docId}/stream?token=${encodeURIComponent(token)}`;
    },
};

export default client;
