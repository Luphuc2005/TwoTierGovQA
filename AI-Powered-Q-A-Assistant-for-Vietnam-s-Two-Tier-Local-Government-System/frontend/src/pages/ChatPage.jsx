/**
 * ChatPage — main chat interface with sidebar, header, and message area.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import ConversationSidebar from "../components/ConversationSidebar";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import UploadModal from "../components/UploadModal";
import ProcessingModal from "../components/ProcessingModal";
import { conversationAPI, chatAPI, uploadAPI } from "../api/client";

export default function ChatPage() {
    const [conversations, setConversations] = useState([]);
    const [activeConversationId, setActiveConversationId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [isProcessingModalOpen, setIsProcessingModalOpen] = useState(false);
    const [processingFile, setProcessingFile] = useState(null);
    const [processingDocId, setProcessingDocId] = useState(null);
    const [uploadError, setUploadError] = useState("");

    // Ref to skip loadMessages when auto-creating a conversation during send
    const skipLoadRef = useRef(false);

    // ── Load conversations on mount ─────────────────────────────
    useEffect(() => {
        loadConversations();
    }, []);

    const loadConversations = async () => {
        try {
            const res = await conversationAPI.list();
            setConversations(res.data.conversations);
        } catch (err) {
            console.error("Failed to load conversations:", err);
        }
    };

    // ── Load messages when active conversation changes ──────────
    useEffect(() => {
        if (activeConversationId) {
            // Skip loading if we just auto-created this conversation during send
            if (skipLoadRef.current) {
                skipLoadRef.current = false;
                return;
            }
            loadMessages(activeConversationId);
        } else {
            setMessages([]);
        }
    }, [activeConversationId]);

    const loadMessages = async (conversationId) => {
        try {
            const res = await chatAPI.getMessages(conversationId);
            setMessages(res.data.messages);
        } catch (err) {
            console.error("Failed to load messages:", err);
        }
    };

    // ── Create new conversation ────────────────────────────────
    const handleNewConversation = async () => {
        try {
            const res = await conversationAPI.create("New Conversation");
            const newConv = res.data;
            setConversations((prev) => [newConv, ...prev]);
            setActiveConversationId(newConv.id);
            setMessages([]);
        } catch (err) {
            console.error("Failed to create conversation:", err);
        }
    };

    // ── Delete conversation ─────────────────────────────────────
    const handleDeleteConversation = async (id) => {
        try {
            await conversationAPI.delete(id);
            setConversations((prev) => prev.filter((c) => c.id !== id));
            if (activeConversationId === id) {
                setActiveConversationId(null);
                setMessages([]);
            }
        } catch (err) {
            console.error("Failed to delete conversation:", err);
        }
    };

    // ── Send message ────────────────────────────────────────────
    const handleSendMessage = useCallback(
        async (text) => {
            let convId = activeConversationId;
            let isNewConversation = false;

            // Auto-create conversation if none is active
            if (!convId) {
                try {
                    const res = await conversationAPI.create("New Conversation");
                    const newConv = res.data;
                    setConversations((prev) => [newConv, ...prev]);
                    convId = newConv.id;
                    isNewConversation = true;
                    // Mark to skip loadMessages when setActiveConversationId triggers useEffect
                    skipLoadRef.current = true;
                    setActiveConversationId(newConv.id);
                } catch (err) {
                    console.error("Failed to create conversation:", err);
                    return;
                }
            }

            // Optimistic UI: add user message immediately
            const tempUserMsg = {
                id: `temp-user-${Date.now()}`,
                conversation_id: convId,
                role: "user",
                content: text,
                created_at: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, tempUserMsg]);
            setIsLoading(true);

            try {
                const res = await chatAPI.sendMessage(convId, text);
                const { user_message, assistant_message } = res.data;

                // Replace optimistic user message with real one, and add assistant message
                setMessages((prev) => [
                    ...prev.filter((m) => m.id !== tempUserMsg.id),
                    user_message,
                    assistant_message,
                ]);

                // Refresh conversation list to get updated titles
                loadConversations();
            } catch (err) {
                console.error("Chat error:", err);
                // Add error message
                const errorMsg = {
                    id: `error-${Date.now()}`,
                    conversation_id: convId,
                    role: "assistant",
                    content:
                        "⚠️ Xin lỗi, đã xảy ra lỗi. Vui lòng thử lại.\n\n" +
                        `*Lỗi: ${err.response?.data?.detail || err.message}*`,
                    created_at: new Date().toISOString(),
                };
                setMessages((prev) => [...prev, errorMsg]);
            } finally {
                setIsLoading(false);
            }
        },
        [activeConversationId]
    );

    // ── Handle PDF Upload ────────────────────────────────────────
    const handleUploadFiles = async (files) => {
        const file = files[0];
        if (!file) return;

        setIsUploadModalOpen(false);
        setProcessingFile(file);
        setUploadError("");

        try {
            // Step 1: Upload the file to backend
            const res = await uploadAPI.upload(file);
            const { doc_id } = res.data;

            // Step 2: Open processing modal with real doc_id
            setProcessingDocId(doc_id);
            setTimeout(() => setIsProcessingModalOpen(true), 200);
        } catch (err) {
            console.error("Upload failed:", err);
            const detail = err.response?.data?.detail || err.message;
            setUploadError(`Tải lên thất bại: ${detail}`);
            // Re-open upload modal to show error
            setIsUploadModalOpen(true);
        }
    };

    return (
        <div className="flex h-screen bg-surface overflow-hidden">
            {/* ── Mobile sidebar toggle ─────────────────── */}
            <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                id="btn-toggle-sidebar"
                className="md:hidden fixed top-4 left-4 z-50 p-2.5 rounded-xl
                 bg-white border border-bdr text-txt-primary
                 shadow-card hover:shadow-card-lg transition-smooth"
                aria-label="Mở/đóng sidebar"
            >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd"
                        d="M2 4.75A.75.75 0 0 1 2.75 4h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 4.75Zm0 10.5a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75ZM2 10a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 10Z"
                        clipRule="evenodd" />
                </svg>
            </button>

            {/* ── Sidebar ───────────────────────────────── */}
            <div
                className={`${
                    sidebarOpen ? "translate-x-0" : "-translate-x-full"
                } md:translate-x-0 fixed md:relative z-40 transition-transform duration-300 ease-in-out`}
            >
                <ConversationSidebar
                    conversations={conversations}
                    activeId={activeConversationId}
                    onSelect={(id) => {
                        setActiveConversationId(id);
                        setSidebarOpen(false);
                    }}
                    onNew={handleNewConversation}
                    onDelete={handleDeleteConversation}
                />
            </div>

            {/* ── Overlay for mobile sidebar ────────────── */}
            {sidebarOpen && (
                <div
                    className="md:hidden fixed inset-0 bg-black/20 z-30 glass-overlay"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* ── Main Chat Area ─────────────────────────── */}
            <main className="flex-1 flex flex-col min-w-0">
                {/* Header Bar */}
                <header className="bg-white border-b border-bdr px-6 py-3.5 flex items-center justify-between shadow-header">
                    <div className="flex items-center gap-3">
                        {/* Mobile: menu button spacer */}
                        <div className="md:hidden w-10" />

                        {/* Header icon */}
                        <div className="w-8 h-8 rounded-lg bg-primary-100 flex items-center justify-center flex-shrink-0">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"
                                className="w-4.5 h-4.5 text-primary-600">
                                <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                                <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                            </svg>
                        </div>

                        <div>
                            <h1 className="text-[15px] font-bold text-txt-primary leading-tight">
                                Trợ Lý Pháp Lý AI
                            </h1>
                            <p className="text-xs text-txt-label leading-tight">
                                Hỗ trợ tra cứu văn bản pháp luật Việt Nam
                            </p>
                        </div>
                    </div>

                    {/* Status badge */}
                    <div className="flex items-center gap-2">
                        <div className={`flex items-center gap-2 px-3.5 py-2 rounded-full text-xs font-semibold
                            ${isLoading
                                ? 'bg-amber-50 text-amber-700 border border-amber-200'
                                : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            }`}>
                            <div className={`w-2 h-2 rounded-full ${
                                isLoading ? 'bg-amber-500 animate-pulse-soft' : 'bg-emerald-500'
                            }`} />
                            {isLoading ? "Đang xử lý..." : "Sẵn sàng"}
                        </div>
                    </div>
                </header>

                <ChatWindow messages={messages} isLoading={isLoading} />
                <ChatInput
                    onSend={handleSendMessage}
                    disabled={isLoading}
                    onOpenUpload={() => setIsUploadModalOpen(true)}
                />
            </main>

            {/* Document Upload Modal */}
            <UploadModal
                isOpen={isUploadModalOpen}
                onClose={() => {
                    setIsUploadModalOpen(false);
                    setUploadError("");
                }}
                onUpload={handleUploadFiles}
                externalError={uploadError}
            />

            {/* Processing Modal */}
            <ProcessingModal
                isOpen={isProcessingModalOpen}
                onClose={() => {
                    setIsProcessingModalOpen(false);
                    setProcessingDocId(null);
                }}
                file={processingFile}
                docId={processingDocId}
            />
        </div>
    );
}
