import style from "./style.module.css";

export default function StartInfo(){
    return <div className={style.container}>
        <img src="/images/robot.svg"/>
        <p>Возникли вопросы? Напишите нам</p>
    </div>
}