import style from "./style.module.css";

export default function Button({children, ...props}){
    return <button {...props} className={style.btn}>
        {children}
    </button>
}