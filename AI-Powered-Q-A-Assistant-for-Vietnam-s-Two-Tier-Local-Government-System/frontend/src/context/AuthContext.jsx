/**
 * Authentication context — manages user state, login, register, logout.
 */
import { createContext, useContext, useState, useEffect } from "react";
import { authAPI } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // On mount, check for existing token and validate it
    useEffect(() => {
        const token = localStorage.getItem("token");
        if (token) {
            authAPI
                .getMe()
                .then((res) => setUser(res.data))
                .catch(() => {
                    localStorage.removeItem("token");
                    setUser(null);
                })
                .finally(() => setLoading(false));
        } else {
            setLoading(false);
        }
    }, []);

    const login = async (email, password) => {
        const res = await authAPI.login(email, password);
        localStorage.setItem("token", res.data.access_token);
        const me = await authAPI.getMe();
        setUser(me.data);
        return me.data;
    };

    const register = async (email, password, fullName) => {
        const res = await authAPI.register(email, password, fullName);
        localStorage.setItem("token", res.data.access_token);
        const me = await authAPI.getMe();
        setUser(me.data);
        return me.data;
    };

    const logout = () => {
        localStorage.removeItem("token");
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, loading, login, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used within AuthProvider");
    return ctx;
}
