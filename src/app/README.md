# Frontend - AI-Driven Motion Predictor

React-based chat interface for interacting with the stock analysis AI agent. Built with modern web technologies and integrated with the Vercel AI SDK for streaming responses.

## Tech Stack

- **Framework**: React 19.2.4 with TypeScript 5.9.3
- **Build Tool**: Vite 7.3.1
- **UI Library**: shadcn/ui with Radix UI primitives
- **Styling**: Tailwind CSS 4.2.1
- **AI Integration**: AI SDK React 3.0.118 for streaming chat
- **Charts**: Recharts 3.8.0
- **Icons**: Lucide React 0.577.0

## Prerequisites

- **Node.js**: Version 20 or higher
- **pnpm**: Package manager (install via `npm install -g pnpm` or see [pnpm.io/installation](https://pnpm.io/installation))

## Installation

1. **Navigate to the app directory**:

   ```sh
   cd src/app
   ```

1. **Install dependencies**:

   ```sh
   pnpm install
   ```

## Environment Variables

Create a `.env` file in the `src/app` directory:

```env
VITE_API_URL=http://localhost:8000
```

The `VITE_API_URL` should point to your backend API server.

## Available Scripts

All scripts are defined in `package.json`:

| Script      | Command          | Description                              |
| ----------- | ---------------- | ---------------------------------------- |
| `dev`       | `pnpm dev`       | Start development server with Vite       |
| `build`     | `pnpm build`     | Type-check and build for production      |
| `preview`   | `pnpm preview`   | Preview the production build locally     |
| `lint`      | `pnpm lint`      | Run ESLint on the codebase               |
| `format`    | `pnpm format`    | Format code with Prettier                |
| `typecheck` | `pnpm typecheck` | Run TypeScript compiler without emitting |

## Development

Start the development server:

```sh
pnpm dev
```

This will start the Vite dev server (default: http://localhost:5173) with hot module replacement enabled.

The frontend expects the backend API to be running at the URL specified in `VITE_API_URL` (default: http://localhost:8000).

## Building for Production

```sh
pnpm build
```

This runs TypeScript compilation (`tsc -b`) followed by Vite build. Output is placed in the `dist/` directory.

Preview the production build:

```sh
pnpm preview
```

## Project Structure

```
src/app/
├── src/
│   ├── components/      # React components
│   ├── components/ui/   # shadcn/ui components
│   ├── hooks/           # Custom React hooks
│   ├── lib/             # Utility functions
│   ├── types/           # TypeScript type definitions
│   ├── App.tsx          # Root application component
│   └── main.tsx         # Application entry point
├── public/              # Static assets
├── index.html           # HTML entry point
├── vite.config.ts       # Vite configuration
├── tsconfig.json        # TypeScript configuration
├── eslint.config.js     # ESLint configuration
└── .prettierrc          # Prettier configuration
```

## Adding shadcn/ui Components

To add new shadcn/ui components:

```sh
npx shadcn@latest add button
```

Components are installed to `src/components/ui/`.

## Key Dependencies

From `package.json`:

**Production:**

- `@ai-sdk/react`: AI SDK React integration for streaming
- `ai`: Vercel AI SDK core
- `react-markdown`: Markdown rendering with `remark-gfm` for GitHub-flavored markdown
- `streamdown`: Streaming markdown renderer
- `@streamdown/code`: Code highlighting for Streamdown
- `class-variance-authority` + `clsx` + `tailwind-merge`: Utility classes management
- `tw-animate-css`: Tailwind CSS animations

**Development:**

- `@vitejs/plugin-react`: React Fast Refresh
- `typescript-eslint`: TypeScript ESLint integration
- `prettier-plugin-tailwindcss`: Prettier plugin for Tailwind class sorting

## CORS Configuration

The frontend communicates with the backend API. Ensure the backend CORS middleware allows requests from your frontend origin:

```python
# In backend main.py - already configured for development:
allow_origins = ["*"]  # Change to specific origin in production
```

## Integration with Backend

The frontend uses the Vercel AI SDK's `useChat` hook to communicate with the backend's `/api/chat` endpoint. The backend streams responses in AI SDK v5 Data Stream Protocol format with:

- `text-start` / `text-delta` / `text-end`: Streaming text content
- `data-tool-call`: Tool invocation events
- `data-tool-result`: Tool execution results
- `data-chart`: Chart data for visualization
- `finish`: Message completion signal

## Troubleshooting

**Port already in use:**
Vite will automatically try the next available port if 5173 is in use.

**API connection errors:**
Verify the backend is running and `VITE_API_URL` is correctly set.

**Type errors:**
Run `pnpm typecheck` to see detailed TypeScript errors.
