import "./index.css";
import LogoNav from "./components/logoNav";
import { useState } from "react";
import MessageInput from "./components/messageInput";
import StartInfo from "./components/startinfo";
import ResultView from "./components/resultView";

function App() {

  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);

  function handleClick(){
    fetch("http://localhost:5000/interaction/askProblem", {
      method: "post",
      body: JSON.stringify({ question: query }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then(setResponse);
  }

  return (
    <>
      <div className="border-2 border-black fixed bottom-0 right-0">
        <aside>
          <LogoNav />
          {
            response
            ? <ResultView response={response}/>
            : <StartInfo/>
          }
          <MessageInput msg={query} setMsg={setQuery} onClick={handleClick}/>
        </aside>
      </div>
    </>
  );
}

export default App;
