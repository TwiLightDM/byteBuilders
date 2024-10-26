import { useState } from "react";

export default function Field() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  function handle() {
    fetch("http://localhost:5000/interaction/askProblem", {
      method: "post",
      body: JSON.stringify({ question: query }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((obj) => setAnswer(obj.answer));
  }
  return (
    <>
      <div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handle}>Нажать</button>
        <p>{answer}</p>
      </div>
    </>
  );
}
