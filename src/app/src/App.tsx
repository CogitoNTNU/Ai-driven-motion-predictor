import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { SearchPage } from "@/pages/SearchPage";
import { DashboardPage } from "@/pages/DashboardPage";

const router = createBrowserRouter([
  { path: "/", element: <SearchPage /> },
  { path: "/stock/:ticker", element: <DashboardPage /> },
]);

export function App() {
  return <RouterProvider router={router} />;
}

export default App;
