import "./index.css";
import Head from "./components/Head";
import LogoNav from "./components/LogoNav";
import Field from "./components/Field";
import { useState } from "react";

function App() {
  return (
    <>
      <div className="border-2 border-black fixed bottom-0 right-0">
        <Head>
          <LogoNav />
          <Field />
        </Head>
      </div>
    </>
  );
}

export default App;
