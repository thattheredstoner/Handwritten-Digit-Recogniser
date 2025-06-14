// Server Configuration Utility
class ServerConfig {
    constructor() {
        this.defaultServer = 'localhost:3030';
        this.currentServer = this.getServerIP();
        this.serverInput = null;
        this.resetButton = null;
        this.statusElement = null;
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupElements());
        } else {
            this.setupElements();
        }
    }

    setupElements() {
        this.serverInput = document.getElementById('server-ip');
        this.resetButton = document.getElementById('reset-server-btn');
        this.statusElement = document.getElementById('server-status');

        if (this.serverInput) {
            this.serverInput.value = this.currentServer;
            this.serverInput.addEventListener('change', () => this.updateServerIP());
            this.serverInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.updateServerIP();
                }
            });
        }

        if (this.resetButton) {
            this.resetButton.addEventListener('click', () => this.resetToDefault());
        }

        // Update the global API_BASE_URL if it exists
        if (typeof window !== 'undefined' && window.API_BASE_URL !== undefined) {
            window.API_BASE_URL = `http://${this.currentServer}/api`;
        }
    }

    getServerIP() {
        return localStorage.getItem('serverIP') || this.defaultServer;
    }

    setServerIP(ip) {
        const cleanIP = ip.trim();
        if (cleanIP) {
            localStorage.setItem('serverIP', cleanIP);
            this.currentServer = cleanIP;
            
            // Update global API_BASE_URL if it exists
            if (typeof window !== 'undefined') {
                window.API_BASE_URL = `http://${cleanIP}/api`;
                
                // Also update the global variable in any existing scripts
                if (window.updateAPIBaseURL) {
                    window.updateAPIBaseURL(`http://${cleanIP}/api`);
                }
            }
            
            this.updateStatus('connecting', 'Connecting...');
            return true;
        }
        return false;
    }

    updateServerIP() {
        if (this.serverInput) {
            const newIP = this.serverInput.value;
            if (this.setServerIP(newIP)) {
                // Trigger a reconnection if there's a global reconnect function
                if (window.reconnectToServer) {
                    window.reconnectToServer();
                }
            } else {
                this.serverInput.value = this.currentServer; // Revert to current value
            }
        }
    }

    resetToDefault() {
        this.setServerIP(this.defaultServer);
        if (this.serverInput) {
            this.serverInput.value = this.defaultServer;
        }
        
        // Trigger a reconnection if there's a global reconnect function
        if (window.reconnectToServer) {
            window.reconnectToServer();
        }
    }

    updateStatus(status, text) {
        if (this.statusElement) {
            this.statusElement.className = `server-status ${status}`;
            this.statusElement.textContent = text || status;
        }
    }

    getAPIBaseURL() {
        return `http://${this.currentServer}/api`;
    }
}

// Create global instance
window.serverConfig = new ServerConfig();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ServerConfig;
}
