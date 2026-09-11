import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 注册Vue编译插件。
export default defineConfig({
  plugins: [vue()],
  // 配置本地开发服务器。
  server: {
    // 将/api请求代理到本地FastAPI。
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
    watch: {
      // 忽略后端与虚拟环境文件变更。
      ignored: ['**/backend/**', '**/.venv/**'],
    },
  },
})
