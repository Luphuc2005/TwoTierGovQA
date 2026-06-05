/**
 * LoginPage — redesigned login form with legal branding.
 */
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            await login(email, password);
            navigate("/");
        } catch (err) {
            setError(err.response?.data?.detail || "Đăng nhập thất bại. Vui lòng thử lại.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-surface bg-pattern px-4">
            <div className="w-full max-w-md animate-slide-up">
                {/* Brand Header */}
                <div className="text-center mb-8">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-600 to-primary-700
                         flex items-center justify-center mx-auto mb-4 shadow-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 text-white">
                            <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                            <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                        </svg>
                    </div>
                    <h1 className="text-2xl font-bold text-txt-primary mb-1">
                        Chào mừng trở lại
                    </h1>
                    <p className="text-txt-secondary text-sm">
                        Đăng nhập để sử dụng Trợ Lý Pháp Lý AI
                    </p>
                </div>

                {/* Form Card */}
                <div className="bg-surface-card rounded-2xl p-8 shadow-card-lg border border-bdr">
                    <form onSubmit={handleSubmit} className="space-y-5">
                        {error && (
                            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm flex items-start gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 flex-shrink-0 mt-0.5">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" clipRule="evenodd" />
                                </svg>
                                {error}
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium text-txt-primary mb-1.5">
                                Email
                            </label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                id="login-email"
                                className="w-full rounded-xl border border-bdr bg-surface px-4 py-3
                                 text-txt-primary placeholder-txt-muted text-[15px]
                                 focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-400
                                 transition-smooth"
                                placeholder="email@example.com"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-txt-primary mb-1.5">
                                Mật khẩu
                            </label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                minLength={6}
                                id="login-password"
                                className="w-full rounded-xl border border-bdr bg-surface px-4 py-3
                                 text-txt-primary placeholder-txt-muted text-[15px]
                                 focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-400
                                 transition-smooth"
                                placeholder="••••••••"
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            id="btn-login"
                            className="w-full rounded-xl bg-primary-600 px-4 py-3 text-white font-semibold text-[15px]
                             hover:bg-primary-700 transition-smooth shadow-md hover:shadow-lg
                             active:scale-[0.98]
                             disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    Đang đăng nhập...
                                </span>
                            ) : (
                                "Đăng nhập"
                            )}
                        </button>
                    </form>

                    <p className="mt-6 text-center text-sm text-txt-secondary">
                        Chưa có tài khoản?{" "}
                        <Link to="/register" className="text-primary-600 hover:text-primary-700 font-semibold transition-smooth">
                            Đăng ký ngay
                        </Link>
                    </p>
                </div>

                {/* Footer */}
                <p className="text-center text-[11px] text-txt-muted mt-6">
                    Trợ Lý Pháp Lý AI — Đại học Sài Gòn (SGU)
                </p>
            </div>
        </div>
    );
}
