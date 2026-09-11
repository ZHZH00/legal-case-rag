import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 负责开发服务器和生产构建；Vue 插件负责解析、编译项目中的 .vue 文件。
export default defineConfig({
  plugins: [vue()],
  // 前端开发时不监听案件数据和 Python 依赖，避免大量无关文件拖慢页面加载。
  server: {
    // 本地开发时把/api请求转发给直接运行的FastAPI后端。
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
    watch: {
      ignored: ['**/backend/**', '**/.venv/**'],
    },
  },
})
