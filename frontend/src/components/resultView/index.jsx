import { useState } from "react"
import Button from "../button"
import style from "./style.module.css"

export default function ResultView({response: {label, percentages, similarTopics, solution}}){
    const [pageN, setPageN] = useState("solution");

    const pages = {
        "solution": <>
            <p className={style.presolution}>Предлагаемый сервис: {label}</p>
            <p className={style.presolution}>Вот решение проблемы:</p>
            <p className={style.solution}>{solution}</p>
        </>,
        
        "similar": <>
            {similarTopics.map((topic, i) => <div className={style[`block-${i % 2 ? "gray" : "yellow"}`]}>
                <p>Проблема: {topic[0]}</p>
                <p className={style["block-second"]}>Решение: {topic[1]}</p>
            </div>)}
        </>,

        "services": <>
            {percentages.map((perc, i) => <div className={style[`block-${i % 2 ? "gray" : "yellow"}`]}>
                <p>Сервис: {perc[0]}</p>
                <p className={style["block-second"]}>Возможность соответствия: {perc[1].toFixed(2)}%</p>
            </div>)}
        </>
    }

    const pButtons = {
        "solution": "Вернуться к решению",
        "services": "Возможные сервисы",
        "similar": "Похожие проблемы",
    }

    return <div className={style.container}>
        
        {pages[pageN]}
    
        <div className={style.buttons}>
            {Object.keys(pButtons).map(key => <>
                {pageN !== key && <Button onClick={()=>setPageN(key)}>{pButtons[key]}</Button>}
            </>)}
            
        </div>

        <img className={style.robot} src="/images/robot2.svg"/>
        
    </div>
}