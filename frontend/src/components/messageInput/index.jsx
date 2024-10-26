import style from "./style.module.css";
export default function MessageInput({msg, setMsg, onClick}){
    return <div className={style.row}>
        <p className={style.mark}>Я</p>

        <div className={style.container}>
            <input
                type="text"
                value={msg}
                onChange={({target: {value}})=>setMsg(value)}
                placeholder="Сообщение"
            />
            <button onClick={onClick}><img src="/images/arrow.svg"/></button>
        </div>        
    </div>
}